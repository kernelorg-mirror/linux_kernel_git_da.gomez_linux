// SPDX-License-Identifier: GPL-2.0

//! Rust XArray implementation.
//!
//! This module implements an extensible array (aka XArray) via the [`XArray`] type. See [`XArray4`]
//! and [`XArray6`] for common configurations.
//!
//! The array can hold values of `T: ForeignOwnable` or bounded integers. Users insert values into
//! the array via the [`Entry`] type. References to items in the tree are handled by the
//! [`BorrowEntry`] type.
//!
//! # Differences from C XArray
//!
//! The C XArray uses RCU for lock-free reads and an internal spinlock for writes. This
//! implementation does not provide internal locking. A mutable reference is required for writes, a
//! shared reference for reads. Locking can be applied externally. RCU support is planned.
//!
//! This implementation allocates during store. `xa_reserve()` and `xas_nomem()` are not yet
//! supported.
//!
//! C `xa_destroy()` frees only internal nodes; callers must free their own stored values. Dropping
//! [`XArray`] also frees all stored entries.
//!
//! C `xa_mk_value()` issues `WARN_ON` if the value exceeds `LONG_MAX`; the left shift overflows,
//! producing a wrong entry.
//! Rust [`Entry::int()`] rejects out-of-range values at compile time via `Bounded`, and
//! [`Entry::try_int()`] returns [`None`] at runtime.
//!
//! C XArray features like search marks, multi-index entries, advanced API, etc. are not yet
//! implemented.
//!
//! # Data structure
//!
//! The XArray is a radix tree that maps `usize` indices to [`Entry`] values. Each node holds an
//! array of `SIZE` slots (where `SIZE = 1 << SHIFT`), and the tree depth is
//! `ceil(usize::BITS / SHIFT)`. Indices are decomposed so that each chunk indexes a slot in a
//! node at the corresponding level.
//!
//! ```text
//!                                 XArray
//!                                    │
//!                                    ▼
//!                           ┌────────────────┐
//!                           │   Root Node    │  Level = levels-1
//!                           │ slots[0..SIZE] │
//!                           └────────────────┘
//!                            /       |       \
//!               ┌───────────┘        │        └───────────┐
//!               ▼                    ▼                    ▼
//!      ┌────────────────┐       ┌────────┐       ┌────────────────┐
//!      │      Node      │       │ Empty  │       │      Node      │
//!      │ slots[0..SIZE] │       └────────┘       │ slots[0..SIZE] │
//!      └────────────────┘                        └────────────────┘
//!        /           \                                   |
//!       ▼             ▼                                  ▼
//!    ...            ...                          ┌────────────────┐
//!                                                │   Leaf Node    │  Level = 0
//!                                                │ slots[0..SIZE] │
//!                                                └────────────────┘
//!                                                 /       |       \
//!                                                ▼        ▼        ▼
//!                                             Entry    Empty    Entry
//!                                             (Int)             (Ptr)
//! ```
//!
//! Index decomposition for [`XArray6`] (SHIFT=6, 64 slots per node, 11 levels on 64-bit):
//!
//! ```text
//!   ┌─────────────────────────────────────────────────────────────────┐
//!   │ index                                                           │
//!   ├─────────┬─────────┬───────────────┬─────────┬─────────┬─────────┤
//!   │ level10 │ level 9 │      ...      │ level 2 │ level 1 │ level 0 │
//!   │ [63:60] │ [59:54] │               │ [17:12] │ [11:6]  │  [5:0]  │
//!   └─────────┴─────────┴───────────────┴─────────┴─────────┴─────────┘
//! ```
//!
//! At each level, the slot offset is: `(index >> (level * SHIFT)) & (SIZE - 1)`.
//!
//! # C API
//!
//! The Rust API enforces entry invariants at compile or construction time. C users must
//! validate at runtime before calling into Rust:
//!
//! - **Int values**: use [`Entry::try_int()`]; reject [`None`] as `-EINVAL`. C has no [`Bounded`],
//!   so the FFI wrapper is the enforcement point.
//! - **Pointers**: validate 4-byte alignment and non-null at runtime.
//!   `const_assert!(T::FOREIGN_ALIGN >= 4)` only covers Rust callers.
//! - **NULL pointers**: dispatch as erase (`xa_store(NULL) == xa_erase`) or reserve
//!   (`XA_FLAGS_ALLOC`) before reaching [`XArray::store()`].
//! - **Error entries**: reject `xa_is_err()` values; Rust would misclassify them as node pointers
//!   (see `Slot` invariants).

use crate::{
    alloc::Flags,
    num::Bounded,
    prelude::*,
    types::ForeignOwnable, //
};
use core::{
    array,
    marker::PhantomData,
    mem, //
};

/// Type alias for [`XArray`] with a shift of 4 and 16 slots per node.
pub type XArray4<T> = XArray<T, 4, 16>;

/// Type alias for [`XArray`] with a shift of 6 and 64 slots per node.
pub type XArray6<T> = XArray<T, 6, 64>;

/// Type alias for [`Entry::Int`] values. Integers that fit in `usize::BITS - 1` bits
/// (`0..=usize::MAX >> 1`).
pub type Value = Bounded<usize, { usize::BITS - 1 }>;

/// An entry is either an [`Entry::Int`] integer (`0..=usize::MAX >> 1`) or an owned
/// [`Entry::Pointer`].
///
/// Empty slots are represented by [`None`], not a separate variant.
///
/// Integer values are validated by [`Value`]. Pointer alignment (`T::FOREIGN_ALIGN >= 4`) is a
/// requirement of the slot encoding, enforced at compile time when an [`XArray`] over `T` is
/// instantiated (see the [`XArray`] invariants); [`Entry`] itself carries no invariants.
pub enum Entry<T: ForeignOwnable> {
    /// Integer value (`0..=usize::MAX >> 1`).
    Int(Value),
    /// Pointer payload. Owned (`T`).
    Pointer(T),
}

impl<T: ForeignOwnable> Entry<T> {
    /// Creates an [`Entry::Int`] validated at compile time.
    ///
    /// Fails to compile if `V > usize::MAX >> 1`.
    pub const fn int<const V: usize>() -> Self {
        Entry::Int(Value::new::<V>())
    }

    /// Creates an [`Entry::Int`] validated at runtime.
    ///
    /// Returns [`None`] if `v > usize::MAX >> 1`.
    pub fn try_int(v: usize) -> Option<Self> {
        Value::try_new(v).map(Entry::Int)
    }
}

/// A borrowed entry returned by [`XArray::load()`], containing either an integer or a borrowed
/// pointer.
pub enum BorrowEntry<'a, T: ForeignOwnable + 'a> {
    /// Integer value (`0..=usize::MAX >> 1`).
    Int(Value),
    /// Pointer payload `T::Borrowed<'_>`.
    Pointer(T::Borrowed<'a>),
}

/// An extensible array backed by a radix tree, mapping `usize` indices to [`Entry`] values.
///
/// `T` must be a [`ForeignOwnable`] whose `FOREIGN_ALIGN` is at least 4; instantiating an
/// [`XArray`] with any other `T` fails to compile (see `Self::validate`).
///
/// # Ownership
///
/// When an [`XArray`] is dropped, all stored [`Entry::Pointer`] entries are freed via
/// [`ForeignOwnable::from_foreign()`]. This differs from the C `xa_destroy()`, which only frees
/// internal nodes and requires callers to free stored pointers themselves. [`Entry::Int`] entries
/// are encoded integers with no backing allocation and need no cleanup.
///
/// # Examples
///
/// ```
/// use kernel::alloc::{flags, KBox};
/// use kernel::rxarray::{BorrowEntry, Entry, XArray6};
///
/// let mut xa = XArray6::<KBox<u64>>::new();
/// assert!(xa.is_empty());
///
/// // Store a pointer entry at index 1.
/// let boxed = KBox::new(0xbeef_u64, flags::GFP_KERNEL)?;
/// xa.store(1, Entry::Pointer(boxed), flags::GFP_KERNEL)?;
/// match xa.load(1) {
///     Some(BorrowEntry::Pointer(val)) => assert_eq!(*val, 0xbeef_u64),
///     _ => panic!("expected Pointer"),
/// }
///
/// // Store a value entry at index 0.
/// let old = xa.store(0, Entry::int::<0xdead>(), flags::GFP_KERNEL)?;
/// assert!(old.is_none());
/// assert!(!xa.is_empty());
///
/// match xa.load(0) {
///     Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xdead),
///     _ => panic!("expected Int"),
/// }
///
/// let old = xa.store(0, Entry::int::<0xcafe>(), flags::GFP_KERNEL)?;
/// match old {
///     Some(Entry::Int(v)) => assert_eq!(v, 0xdead),
///     _ => panic!("expected old Int"),
/// }
///
/// match xa.erase(0) {
///     Some(Entry::Int(v)) => assert_eq!(v, 0xcafe),
///     _ => panic!("expected erased Int"),
/// }
/// assert!(xa.erase(1).is_some());
/// assert!(xa.is_empty());
///
/// # Ok::<(), Error>(())
/// ```
///
/// # Invariants
///
/// - `T::FOREIGN_ALIGN >= 4`, ensuring pointer entries do not collide with the integer or internal
///   entry encoding (enforced at compile time by `validate()`).
/// - `SHIFT > 0` and `SIZE == 1 << SHIFT` (enforced at compile time by `validate()`).
/// - Every slot in every node satisfies the `Slot` invariants.
/// - Interior levels (level > 0) contain only empty or node slots. Leaf level (level 0) contains
///   only empty, `Int`, or `Pointer` slots.
pub struct XArray<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> {
    root: Node<T, SHIFT, SIZE>,
}

impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> Default for XArray<T, SHIFT, SIZE> {
    fn default() -> Self {
        Self::validate();
        // INVARIANT:
        // - `Self::validate` checks `T::FOREIGN_ALIGN >= 4`, `SHIFT > 0` and `SIZE == 1 << SHIFT`
        //   at compile time.
        // - `Node::new` fills every slot with `Slot::EMPTY`, which satisfies the empty-slot case
        //   of the `Slot` invariants.
        // - Every slot is empty, and an empty slot is valid at any level.
        XArray { root: Node::new() }
    }
}

impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> XArray<T, SHIFT, SIZE> {
    fn validate() {
        const_assert!(SHIFT > 0, "SHIFT must be > 0");
        const_assert!(SIZE == (1 << SHIFT), "SIZE != 1 << SHIFT");
        const_assert!(
            T::FOREIGN_ALIGN >= 4,
            "ForeignOwnable pointers must be 4-byte aligned"
        );
    }

    /// Creates a new empty [`XArray`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of levels in the tree.
    const fn levels() -> usize {
        (usize::BITS as usize).div_ceil(SHIFT)
    }

    /// Returns `true` if the tree contains no entries.
    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Erases the entry at `index`.
    ///
    /// Returns the previous entry, or [`None`] if the slot was empty. Empty intermediate nodes are
    /// freed during traversal.
    pub fn erase(&mut self, index: usize) -> Option<Entry<T>> {
        self.root.erase(index, Self::levels() - 1)
    }

    /// Stores an entry at `index`.
    ///
    /// Returns the previous entry, or [`None`] if the slot was empty.
    ///
    /// # Errors
    ///
    /// Returns [`ENOMEM`] if a new intermediate node cannot be allocated.
    pub fn store(
        &mut self,
        index: usize,
        entry: Entry<T>,
        flags: Flags,
    ) -> Result<Option<Entry<T>>> {
        self.root.store(index, entry, Self::levels() - 1, flags)
    }

    /// Loads the entry at `index`.
    ///
    /// Returns a borrowed view of the entry, or [`None`] if the slot is empty. Pointer entries
    /// are returned as `T::Borrowed<'_>` (e.g., `&T` for `KBox<T>`). Integer entries are returned
    /// as [`Value`].
    pub fn load(&self, index: usize) -> Option<BorrowEntry<'_, T>> {
        self.root.load(index, Self::levels() - 1)
    }
}

/// The internal storage unit: a single `usize` encoding an [`Entry`] or an internal node pointer.
///
/// The `PhantomData<T>` marks [`Slot<T>`] as owning the `T` whose
/// [`ForeignOwnable::into_foreign()`] pointer it encodes.
///
/// # Invariants
///
/// The encoded value `self.0` is one of:
/// - `0`: the slot is empty.
/// - An odd value `(v << 1) | 1`: an integer value `v` where `v <= usize::MAX >> 1`.
/// - An even, non-zero value with bits `1:0 == 0b00`: a valid, non-null pointer previously returned
///   by [`ForeignOwnable::into_foreign()`].
/// - An even, non-zero value with bits `1:0 == 0b10` and value > `NODE_THRESHOLD`: a valid pointer
///   to a live `Node` allocation, created by [`Slot::mk_node()`] from [`KBox::into_raw()`] tagged
///   with `| 2`. The slot owns the [`KBox<Node>`] allocation.
///
/// Values with bits `1:0 == 0b10` with value <= `NODE_THRESHOLD` (reserved for internal entries)
/// are never stored in a slot.
#[repr(transparent)]
struct Slot<T: ForeignOwnable>(usize, PhantomData<T>);

impl<T: ForeignOwnable> Default for Slot<T> {
    fn default() -> Self {
        Self::EMPTY
    }
}

#[derive(Copy, Clone)]
enum SlotType {
    Empty,
    Pointer,
    Internal,
    Node, // Internal with value > NODE_THRESHOLD.
    Int,
}

impl<T: ForeignOwnable> Slot<T> {
    // INVARIANT: `0` is the encoding for an empty slot.
    const EMPTY: Self = Slot(0, PhantomData);

    /// Encodes an [`Entry<T>`] into a [`Slot`] for tree storage.
    fn encode(entry: Entry<T>) -> Result<Self> {
        const_assert!(
            T::FOREIGN_ALIGN >= 4,
            "ForeignOwnable pointers must be 4-byte aligned"
        );

        match entry {
            Entry::Int(v) => {
                // INVARIANT: `*v <= usize::MAX >> 1` is guaranteed by `Value` (`Bounded`), so the
                // shift cannot overflow and the encoded value is odd.
                Ok(Slot((*v << 1) | 1, PhantomData))
            }
            Entry::Pointer(p) => {
                let bits = p.into_foreign() as usize;
                debug_assert!(bits != 0, "ForeignOwnable returned null");
                debug_assert!(bits & 3 == 0, "ForeignOwnable pointer not 4-byte aligned");
                // INVARIANT: `into_foreign()` guarantees a non-null pointer aligned to
                // `T::FOREIGN_ALIGN`, and `const_assert!(T::FOREIGN_ALIGN >= 4)` above ensures
                // bits `1:0 == 0b00`.
                Ok(Slot(bits, PhantomData))
            }
        }
    }

    /// Decodes a [`Slot`] into an owned [`Entry<T>`], consuming the slot.
    ///
    /// Returns [`None`] for empty slots.
    fn decode(self) -> Option<Entry<T>> {
        let slot_type = self.kind();
        let bits = mem::ManuallyDrop::new(self).0;

        match slot_type {
            SlotType::Empty => None,
            // Undo the `(v << 1) | 1` encoding from `Slot::encode`.
            SlotType::Int => Some(Entry::Int(Value::from_expr(bits >> 1))),
            SlotType::Node | SlotType::Internal => {
                debug_assert!(false, "attempt to decode internal/node entry");
                None
            }
            SlotType::Pointer => {
                // SAFETY:
                // - By the type invariant, a slot classified `Pointer` holds a pointer returned by
                //   a previous call to `T::into_foreign()`.
                // - `decode` takes the slot by value, so the caller has already removed it from
                //   the tree, and `ManuallyDrop` suppresses `Slot::drop`. This is therefore the
                //   only `from_foreign` call for this pointer.
                Some(Entry::Pointer(unsafe {
                    T::from_foreign(bits as *mut c_void)
                }))
            }
        }
    }

    /// Borrows the entry in this slot without consuming it.
    ///
    /// Returns [`None`] for empty slots. Pointer entries are borrowed via
    /// [`ForeignOwnable::borrow()`].
    fn borrow(&self) -> Option<BorrowEntry<'_, T>> {
        let slot_type = self.kind();
        let bits = self.0;

        match slot_type {
            SlotType::Empty => None,
            // Undo the `(v << 1) | 1` encoding from `Slot::encode`.
            SlotType::Int => Some(BorrowEntry::Int(Value::from_expr(bits >> 1))),
            SlotType::Node | SlotType::Internal => {
                debug_assert!(false, "attempt to borrow internal/node entry");
                None
            }
            SlotType::Pointer => {
                // SAFETY:
                // - By the type invariant, a slot classified `Pointer` holds a pointer returned by
                //   a previous call to `T::into_foreign()`.
                // - Every path that reaches `from_foreign` for this slot needs ownership of it
                //   (`decode`) or a unique borrow (`Slot::drop`), and reaching either from the
                //   array requires `&mut self` on the `XArray`. Neither can coexist with this
                //   shared borrow, so any `from_foreign` on this pointer happens after the borrow
                //   ends.
                Some(BorrowEntry::Pointer(unsafe {
                    T::borrow(bits as *mut c_void)
                }))
            }
        }
    }

    // All entries with bits `1:0 == 0b10` are internal to the XArray implementation. The encoded
    // value distinguishes sub-types:
    //
    // - Offset 0..=62, encoded 2..=250: sibling entries.
    //   The encoded value contains the offset of the canonical slot within the same node
    //   (multi-index entries, CONFIG_XARRAY_MULTI).
    //
    // - Logical 256, encoded 1026: retry (XA_RETRY_ENTRY).
    //   Tombstone signaling concurrent tree modification; RCU lock-free readers must restart.
    //
    // - Logical 257, encoded 1030: zero (XA_ZERO_ENTRY).
    //   Placeholder marking a slot as occupied but logically empty (xa_reserve,
    //   XA_FLAGS_TRACK_FREE).
    //
    // - Encoded value > 4096: node pointers (heap addresses tagged with | 2).
    //   Always above this threshold.
    //
    // Note: C error entries (xa_is_err()) also encode as internal entries with values far above
    // this threshold. They are never stored in tree slots, only returned by the XArray state
    // machine API. C callers bypass this assumption; the FFI layer must reject error entries before
    // they reach Rust, where kind() would misclassify them as Node, causing as_node_ref() to
    // dereference an invalid pointer.
    //
    // This implementation only creates node pointers via `mk_node()`. The other sub-types are
    // reserved for future RCU and multi-index support. See include/linux/xarray.h.
    const NODE_THRESHOLD: usize = 4096;

    #[inline]
    fn kind(&self) -> SlotType {
        let slot = self.0;
        if slot == 0 {
            return SlotType::Empty;
        }
        match slot & 3 {
            2 if slot > Self::NODE_THRESHOLD => SlotType::Node,
            2 => SlotType::Internal,
            1 | 3 => SlotType::Int,
            _ => SlotType::Pointer,
        }
    }

    #[inline]
    fn is_empty(&self) -> bool {
        matches!(self.kind(), SlotType::Empty)
    }

    #[inline]
    fn is_node(&self) -> bool {
        matches!(self.kind(), SlotType::Node)
    }

    /// Returns a shared reference to the node pointed to by this entry.
    ///
    /// # Safety
    ///
    /// - `self.is_node()` must return `true`.
    /// - The underlying allocation must remain live for the duration of the borrow.
    /// - No mutable reference to the same node allocation must exist for the duration of the
    ///   returned borrow.
    /// - The const generic parameters must match the node type stored in the entry.
    unsafe fn as_node_ref<const SHIFT: usize, const SIZE: usize>(&self) -> &Node<T, SHIFT, SIZE> {
        debug_assert!(self.is_node(), "Not a node entry");
        let ptr = (self.0 & !3) as *const Node<T, SHIFT, SIZE>;
        // SAFETY: By function safety requirements and `Slot` type invariant, a `Node`-typed slot
        // holds a valid, live `KBox` allocation. `& !3` reverses the `| 2` tag.
        unsafe { &*ptr }
    }

    /// Returns a mutable reference to the node pointed to by this entry.
    ///
    /// # Safety
    ///
    /// - `self.is_node()` must return `true`.
    /// - The underlying allocation must remain live for the duration of the borrow.
    /// - The caller must guarantee exclusive access to the node: no other references (shared or
    ///   mutable) to the same node allocation must exist.
    /// - The slot containing this entry must not be modified (e.g., via `mem::take` or
    ///   `mem::replace`) while the returned reference is live.
    /// - The const generic parameters must match the node type stored in the entry.
    unsafe fn as_node_mut<const SHIFT: usize, const SIZE: usize>(
        &mut self,
    ) -> &mut Node<T, SHIFT, SIZE> {
        debug_assert!(self.is_node(), "Not a node entry");
        let ptr = (self.0 & !3) as *mut Node<T, SHIFT, SIZE>;
        // SAFETY: By function safety requirements and `Slot` type invariant, a `Node`-typed slot
        // holds a valid, live `KBox` allocation. `& !3` reverses the `| 2` tag.
        unsafe { &mut *ptr }
    }

    /// Consumes this entry and returns the owned node allocation.
    ///
    /// # Safety
    ///
    /// - `self.is_node()` must return `true`.
    /// - The raw pointer encoded in the entry must be a valid `KBox<Node>` allocation. The caller
    ///   takes ownership of it.
    /// - The const generic parameters must match the node type stored in the entry.
    unsafe fn into_node<const SHIFT: usize, const SIZE: usize>(self) -> KBox<Node<T, SHIFT, SIZE>> {
        debug_assert!(self.is_node(), "Not a node entry");
        let this = mem::ManuallyDrop::new(self);
        let ptr = (this.0 & !3) as *mut Node<T, SHIFT, SIZE>;
        // SAFETY: By function safety requirements and `Slot` type invariant, a `Node`-typed slot
        // holds a valid, live `KBox` allocation. `& !3` reverses the `| 2` tag. `ManuallyDrop`
        // prevents double-free via `Slot::drop`.
        unsafe { KBox::from_raw(ptr) }
    }

    /// Creates a node slot from a heap-allocated [`Node`].
    fn mk_node<const SHIFT: usize, const SIZE: usize>(node: KBox<Node<T, SHIFT, SIZE>>) -> Self {
        let ptr = KBox::into_raw(node) as usize;
        debug_assert!(ptr & 3 == 0, "Node pointer not aligned");
        // INVARIANT: `ptr` owns a live `KBox<Node>` allocation from `into_raw()`, so it is at least
        // 4-byte aligned (checked above) and `| 2` gives bits `1:0 == 0b10`. Kernel heap addresses
        // are far above `NODE_THRESHOLD`, so the result classifies as node.
        Slot(ptr | 2, PhantomData)
    }
}

impl<T: ForeignOwnable> Drop for Slot<T> {
    fn drop(&mut self) {
        match self.kind() {
            SlotType::Pointer => {
                // SAFETY:
                // - By the type invariant, a slot classified `Pointer` holds a pointer returned by
                //   a previous call to `T::into_foreign()`.
                // - Every other `from_foreign` path (`decode`) consumes the slot through
                //   `ManuallyDrop`, which suppresses this destructor, so this is the only
                //   `from_foreign` call for this pointer.
                drop(unsafe { T::from_foreign(self.0 as *mut c_void) });
            }
            SlotType::Node | SlotType::Internal => {
                // Intermediate tree nodes (heap-allocated `KBox<Node>` behind a tagged pointer)
                // are freed by `Node::drop`, which `mem::take`s each child slot and calls
                // `into_node()`. If one reaches here, leak it: `Slot` lacks the const generics to
                // reconstruct the `KBox`.
                pr_warn!("Slot dropped while containing node pointer, memory leak\n");
                debug_assert!(false, "internal/node entry must not reach Slot::drop");
            }
            _ => {}
        }
    }
}

/// A single node in the radix tree. Each node contains `SIZE` slots, where each slot may be empty,
/// contain a user entry (integer or pointer), or point to a child node at the next tree level.
///
/// # Invariants
///
/// - Interior nodes (level > 0): each slot is either `Empty` or a `Node` pointer created by
///   [`Slot::mk_node()`] from a valid [`KBox<Node<T, SHIFT, SIZE>>`] allocation.
/// - Leaf nodes (level == 0): each slot is either `Empty`, `Int`, or `Pointer`. No `Node` or
///   `Internal` slots may appear at level 0.
struct Node<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> {
    slots: [Slot<T>; SIZE],
}

impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> Default for Node<T, SHIFT, SIZE> {
    fn default() -> Self {
        // INVARIANT: every slot is `Slot::EMPTY`, so the node satisfies the type invariants at any
        // level.
        Self {
            slots: array::from_fn(|_| Slot::EMPTY),
        }
    }
}

impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> Node<T, SHIFT, SIZE> {
    // Bitmask for extracting the slot offset. Equivalent to XA_CHUNK_MASK.
    const MASK: usize = SIZE - 1;

    fn new() -> Self {
        Self::default()
    }

    /// Extracts the slot index for `index` at the given tree `level`.
    ///
    /// Equivalent to `get_offset()` (see `lib/xarray.c`).
    #[inline]
    fn slot_index(index: usize, level: usize) -> usize {
        (index >> (level * SHIFT)) & Self::MASK
    }

    /// Returns `true` if every slot in this node is empty.
    fn is_empty(&self) -> bool {
        self.slots.iter().all(|slot| slot.is_empty())
    }

    /// Erases the entry at `index`.
    ///
    /// Returns the previous entry, or [`None`] if the slot was empty. Frees empty intermediate
    /// nodes on the way back up.
    fn erase(&mut self, index: usize, level: usize) -> Option<Entry<T>> {
        let slot_index = Self::slot_index(index, level);
        let slot = &mut self.slots[slot_index];
        if level == 0 {
            debug_assert!(!slot.is_node(), "Found node pointer at leaf level");
            // INVARIANT: `mem::take` leaves `Slot::EMPTY` behind, which is permitted at leaf level.
            let old = mem::take(slot);
            return old.decode();
        }

        if slot.is_node() {
            // SAFETY:
            // - `slot.is_node()` returned true.
            // - By the `Node` type invariant, an interior-level node slot holds a valid, live
            //   `KBox<Node>` allocation.
            // - `&mut self` guarantees exclusive access, and the borrow of `child` ends before the
            //   `mem::take` below.
            // - `SHIFT` and `SIZE` are this node's own const parameters, so they match the child
            //   node type stored in the slot.
            let child = unsafe { slot.as_node_mut::<SHIFT, SIZE>() };
            let old_value = child.erase(index, level - 1);
            if child.is_empty() {
                // INVARIANT: `mem::take` leaves `Slot::EMPTY` behind, which is permitted at
                // interior level.
                let old_slot = mem::take(slot);
                // SAFETY:
                // - `is_node()` was checked above and is preserved by `mem::take`.
                // - By the `Node` type invariant, the slot holds a valid `KBox<Node>` allocation,
                //   whose ownership passes to `child_box`.
                // - `SHIFT` and `SIZE` are this node's own const parameters, so they match the
                //   child node type stored in the slot.
                let child_box: KBox<Node<T, SHIFT, SIZE>> = unsafe { old_slot.into_node() };
                drop(child_box);
            }
            old_value
        } else {
            // Values at intermediate levels indicate tree corruption.
            debug_assert!(slot.is_empty(), "Non-null non-node entry");
            None
        }
    }

    /// Stores `entry` at `index`, allocating intermediate nodes as needed.
    ///
    /// Returns the previous entry, or [`None`] if the slot was empty.
    fn store(
        &mut self,
        index: usize,
        entry: Entry<T>,
        level: usize,
        flags: Flags,
    ) -> Result<Option<Entry<T>>> {
        let slot_index = Self::slot_index(index, level);
        let slot = &mut self.slots[slot_index];
        if level == 0 {
            debug_assert!(!slot.is_node(), "Found node pointer at leaf level");
            // INVARIANT: `Slot::encode` returns an `Int` or `Pointer` slot, both of which are
            // permitted at leaf level.
            let new_slot = Slot::encode(entry)?;
            let old = mem::replace(slot, new_slot);
            return Ok(old.decode());
        }

        if slot.is_empty() {
            let child_node = KBox::new(Self::new(), flags)?;
            // INVARIANT: `Slot::mk_node` returns a node slot holding a valid `KBox<Node>`, which
            // is permitted at interior level.
            *slot = Slot::mk_node(child_node);
        }

        if slot.is_node() {
            // SAFETY:
            // - `slot.is_node()` returned true.
            // - By the `Node` type invariant, an interior-level node slot holds a valid, live
            //   `KBox<Node>` allocation.
            // - `&mut self` guarantees exclusive access, and the slot is not modified while the
            //   returned reference is live.
            // - `SHIFT` and `SIZE` are this node's own const parameters, so they match the child
            //   node type stored in the slot.
            let child_node = unsafe { slot.as_node_mut::<SHIFT, SIZE>() };
            child_node.store(index, entry, level - 1, flags)
        } else {
            // Unreachable unless the tree is corrupt. Rust type safety prevents this; C callers
            // bypass that guarantee, so the FFI layer must ensure tree integrity before reaching
            // this path.
            debug_assert!(false, "Non-node slot after node creation");
            Err(EINVAL)
        }
    }

    /// Loads the entry at `index`.
    ///
    /// Returns a borrowed view of the entry, or [`None`] if the slot is empty.
    fn load(&self, index: usize, level: usize) -> Option<BorrowEntry<'_, T>> {
        let slot_index = Self::slot_index(index, level);
        let slot = &self.slots[slot_index];
        if level == 0 {
            debug_assert!(!slot.is_node(), "Found node pointer at leaf level");
            return slot.borrow();
        }

        if slot.is_node() {
            // SAFETY:
            // - `slot.is_node()` returned true.
            // - By the `Node` type invariant, an interior-level node slot holds a valid, live
            //   `KBox<Node>` allocation.
            // - `&self` is a shared borrow, so no mutable reference to the node can exist for the
            //   duration of the returned borrow.
            // - `SHIFT` and `SIZE` are this node's own const parameters, so they match the child
            //   node type stored in the slot.
            let child_node = unsafe { slot.as_node_ref::<SHIFT, SIZE>() };
            child_node.load(index, level - 1)
        } else {
            // Values at intermediate levels indicate tree corruption.
            debug_assert!(slot.is_empty(), "Non-null non-node entry");
            None
        }
    }
}

impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize> Drop for Node<T, SHIFT, SIZE> {
    fn drop(&mut self) {
        for slot in &mut self.slots {
            if slot.is_node() {
                // INVARIANT: `mem::take` leaves `Slot::EMPTY` behind, which is permitted at every
                // level.
                let old = mem::take(slot);
                // SAFETY:
                // - `is_node()` was checked above and is preserved by `mem::take`.
                // - By the `Node` type invariant, the slot holds a valid `KBox<Node>` allocation,
                //   whose ownership passes to `node`.
                // - `SHIFT` and `SIZE` are this node's own const parameters, so they match the
                //   child node type stored in the slot.
                let node: KBox<Node<T, SHIFT, SIZE>> = unsafe { old.into_node() };
                drop(node);
            }
        }
        // Remaining slots (Empty, Int, Pointer) are dropped by Slot::drop.
    }
}

#[macros::kunit_tests(rust_rxarray)]
mod tests {
    use super::*;
    use kernel::alloc::flags;

    // Dispatches a test across all SHIFT/SIZE configurations. Default arm passes <T, SHIFT, SIZE>
    // with T = KBox<u64>; `ptr` arm passes <SHIFT, SIZE> only for pointer tests that must construct
    // KBox<u64> values because ForeignOwnable has no constructor, so T must be concrete.
    macro_rules! for_each_xarray {
        ($fn:ident) => {
            $fn::<KBox<u64>, 4, 16>();
            $fn::<KBox<u64>, 6, 64>();
        };
        (ptr, $fn:ident) => {
            $fn::<4, 16>();
            $fn::<6, 64>();
        };
    }

    // `XArray` carries no explicit `Send`/`Sync` impl: the `PhantomData<T>` in `Slot` makes the
    // auto-derived bounds follow `T`. These assertions fail to compile if that ever stops holding.
    const fn assert_send<T: Send>() {}
    const fn assert_sync<T: Sync>() {}

    fn assert_auto_traits_impl<
        T: ForeignOwnable + Send + Sync,
        const SHIFT: usize,
        const SIZE: usize,
    >() {
        assert_send::<XArray<T, SHIFT, SIZE>>();
        assert_sync::<XArray<T, SHIFT, SIZE>>();
    }

    #[test]
    fn assert_auto_traits() {
        for_each_xarray!(assert_auto_traits_impl);
    }

    fn new_is_empty_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let xa = XArray::<T, SHIFT, SIZE>::new();
        assert!(xa.is_empty());
        assert!(xa.load(0).is_none());
        assert!(xa.load(usize::MAX).is_none());
    }

    #[test]
    fn new_is_empty() {
        for_each_xarray!(new_is_empty_impl);
    }

    fn store_load_value_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        let old = xa.store(0, Entry::int::<137>(), flags::GFP_KERNEL).unwrap();
        assert!(old.is_none());
        assert!(!xa.is_empty());

        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 137),
            _ => panic!("expected Int"),
        };
    }

    #[test]
    fn store_load_value() {
        for_each_xarray!(store_load_value_impl);
    }

    fn overwrite_value_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(137, Entry::int::<1001>(), flags::GFP_KERNEL)
            .unwrap();
        let old = xa
            .store(137, Entry::int::<2002>(), flags::GFP_KERNEL)
            .unwrap();

        match old {
            Some(Entry::Int(v)) => assert_eq!(v, 1001),
            _ => panic!("expected old Int"),
        }
        match xa.load(137) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 2002),
            _ => panic!("expected Int"),
        };
    }

    #[test]
    fn overwrite_value() {
        for_each_xarray!(overwrite_value_impl);
    }

    fn erase_value_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(137, Entry::int::<1001>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(138, Entry::int::<1002>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(139, Entry::int::<1003>(), flags::GFP_KERNEL)
            .unwrap();

        let old = xa.erase(138);
        match old {
            Some(Entry::Int(v)) => assert_eq!(v, 1002),
            _ => panic!("expected erased Int"),
        }
        assert!(xa.load(138).is_none());

        match xa.load(137) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 1001),
            _ => panic!("expected Int at neighbor"),
        }
        match xa.load(139) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 1003),
            _ => panic!("expected Int at neighbor"),
        };
    }

    #[test]
    fn erase_value() {
        for_each_xarray!(erase_value_impl);
    }

    // Two paths: (1) empty tree with no nodes allocated, (2) intermediate nodes exist but the
    // target leaf slot was never stored.
    fn erase_nonexistent_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        assert!(xa.erase(999).is_none());

        // Intermediate nodes exist for index 0; index 1 shares the same leaf node but its slot was
        // never populated.
        xa.store(0, Entry::int::<1>(), flags::GFP_KERNEL).unwrap();
        assert!(xa.erase(1).is_none());
        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 1),
            _ => panic!("expected Int after erasing neighbor"),
        };
    }

    #[test]
    fn erase_nonexistent() {
        for_each_xarray!(erase_nonexistent_impl);
    }

    // After erase frees intermediate nodes, re-store at the same index must re-allocate them.
    fn erase_and_restore_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(137, Entry::int::<1001>(), flags::GFP_KERNEL)
            .unwrap();

        let erased = xa.erase(137);
        match erased {
            Some(Entry::Int(v)) => assert_eq!(v, 1001),
            _ => panic!("expected erased Int"),
        }
        assert!(xa.load(137).is_none());

        let old = xa
            .store(137, Entry::int::<2002>(), flags::GFP_KERNEL)
            .unwrap();
        assert!(old.is_none());
        match xa.load(137) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 2002),
            _ => panic!("expected Int"),
        };
    }

    #[test]
    fn erase_and_restore() {
        for_each_xarray!(erase_and_restore_impl);
    }

    // Stores at node boundaries to exercise tree structure:
    // - SIZE-1: last slot in the first leaf node.
    // - SIZE: first index requiring a second leaf node.
    // - SIZE*SIZE-1: last index in the first level-1 subtree.
    // - SIZE*SIZE: first index requiring a third tree level.
    fn store_at_boundaries_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(SIZE - 1, Entry::int::<0xA>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE, Entry::int::<0xB>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE * SIZE - 1, Entry::int::<0xC>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE * SIZE, Entry::int::<0xD>(), flags::GFP_KERNEL)
            .unwrap();

        match xa.load(SIZE - 1) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xA),
            _ => panic!("expected Int at SIZE-1"),
        }
        match xa.load(SIZE) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xB),
            _ => panic!("expected Int at SIZE"),
        }
        match xa.load(SIZE * SIZE - 1) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xC),
            _ => panic!("expected Int at SIZE*SIZE-1"),
        }
        match xa.load(SIZE * SIZE) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xD),
            _ => panic!("expected Int at SIZE*SIZE"),
        }
        assert!(xa.load(0).is_none());
    }

    #[test]
    fn store_at_boundaries() {
        for_each_xarray!(store_at_boundaries_impl);
    }

    // For XArray6 this allocates 10 intermediate nodes; for XArray4, 15. Erasing verifies cascading
    // cleanup frees all intermediate nodes back to the root.
    fn store_at_max_index_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(usize::MAX, Entry::int::<0xFF>(), flags::GFP_KERNEL)
            .unwrap();
        match xa.load(usize::MAX) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0xFF),
            _ => panic!("expected Int at usize::MAX"),
        }
        assert!(xa.load(0).is_none());

        match xa.erase(usize::MAX) {
            Some(Entry::Int(v)) => assert_eq!(v, 0xFF),
            _ => panic!("expected erased Int at usize::MAX"),
        }
        assert!(xa.load(usize::MAX).is_none());
        assert!(xa.is_empty());
    }

    #[test]
    fn store_at_max_index() {
        for_each_xarray!(store_at_max_index_impl);
    }

    // SIZE*3 entries spanning multiple leaf nodes. Erases even indices and verifies odd indices
    // remain intact. Erase return values are not checked here; that path is covered by erase_value.
    fn bulk_operations_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        let count = SIZE * 3;

        for i in 0..count {
            xa.store(i, Entry::try_int(i * 10).unwrap(), flags::GFP_KERNEL)
                .unwrap();
        }

        for i in 0..count {
            if i % 2 == 0 {
                xa.erase(i);
            }
        }

        for i in 0..count {
            match xa.load(i) {
                Some(BorrowEntry::Int(v)) => {
                    assert_eq!(i % 2, 1);
                    assert_eq!(v, i * 10);
                }
                Some(BorrowEntry::Pointer(_)) => {
                    panic!("unexpected Pointer in value-only tree")
                }
                None => assert_eq!(i % 2, 0),
            }
        }
    }

    #[test]
    fn bulk_operations() {
        for_each_xarray!(bulk_operations_impl);
    }

    // Stores entries in different subtrees (indices 0 and SIZE diverge at level 1), then erases
    // both. The second erase cascades cleanup through intermediate nodes that no longer have any
    // children.
    fn erase_all_restores_empty_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(0, Entry::int::<1>(), flags::GFP_KERNEL).unwrap();
        xa.store(SIZE, Entry::int::<2>(), flags::GFP_KERNEL)
            .unwrap();
        assert!(!xa.is_empty());

        xa.erase(0);
        assert!(!xa.is_empty());
        xa.erase(SIZE);
        assert!(xa.is_empty());
    }

    #[test]
    fn erase_all_restores_empty() {
        for_each_xarray!(erase_all_restores_empty_impl);
    }

    // Entry::int::<0>() encodes as (0 << 1) | 1 = 1; must not be confused with the empty slot
    // encoding (0).
    fn value_zero_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(0, Entry::int::<0>(), flags::GFP_KERNEL).unwrap();
        assert!(!xa.is_empty());
        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 0),
            _ => panic!("expected Int"),
        }

        match xa.erase(0) {
            Some(Entry::Int(v)) => assert_eq!(v, 0),
            _ => panic!("expected erased Int"),
        }
        assert!(xa.load(0).is_none());
    }

    #[test]
    fn value_zero() {
        for_each_xarray!(value_zero_impl);
    }

    // Pointer-specific tests (ForeignOwnable path with KBox<u64>).

    fn store_load_pointer_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let boxed = KBox::new(42u64, flags::GFP_KERNEL).unwrap();
        let old = xa
            .store(0, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();
        assert!(old.is_none());

        match xa.load(0) {
            Some(BorrowEntry::Pointer(val)) => assert_eq!(*val, 42u64),
            _ => panic!("expected Pointer"),
        }
    }

    #[test]
    fn store_load_pointer() {
        for_each_xarray!(ptr, store_load_pointer_impl);
    }

    // Erase returns the owned KBox, verifying that Slot::decode transfers ownership via
    // ManuallyDrop without double-free or leak.
    fn erase_returns_pointer_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let boxed = KBox::new(99u64, flags::GFP_KERNEL).unwrap();
        xa.store(5, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();

        match xa.erase(5) {
            Some(Entry::Pointer(owned)) => assert_eq!(*owned, 99u64),
            _ => panic!("expected owned Pointer"),
        }
        assert!(xa.load(5).is_none());
    }

    #[test]
    fn erase_returns_pointer() {
        for_each_xarray!(ptr, erase_returns_pointer_impl);
    }

    fn overwrite_pointer_with_value_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let boxed = KBox::new(42u64, flags::GFP_KERNEL).unwrap();
        xa.store(0, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();

        let old = xa.store(0, Entry::int::<137>(), flags::GFP_KERNEL).unwrap();
        match old {
            Some(Entry::Pointer(p)) => assert_eq!(*p, 42u64),
            _ => panic!("expected old Pointer"),
        }
        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 137),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn overwrite_pointer_with_value() {
        for_each_xarray!(ptr, overwrite_pointer_with_value_impl);
    }

    fn overwrite_value_with_pointer_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        xa.store(0, Entry::int::<137>(), flags::GFP_KERNEL).unwrap();

        let boxed = KBox::new(42u64, flags::GFP_KERNEL).unwrap();
        let old = xa
            .store(0, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();
        match old {
            Some(Entry::Int(v)) => assert_eq!(v, 137),
            _ => panic!("expected old Int"),
        }
        match xa.load(0) {
            Some(BorrowEntry::Pointer(val)) => assert_eq!(*val, 42u64),
            _ => panic!("expected Pointer"),
        }
    }

    #[test]
    fn overwrite_value_with_pointer() {
        for_each_xarray!(ptr, overwrite_value_with_pointer_impl);
    }

    fn mixed_values_and_pointers_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let boxed = KBox::new(42u64, flags::GFP_KERNEL).unwrap();
        xa.store(0, Entry::int::<100>(), flags::GFP_KERNEL).unwrap();
        xa.store(1, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();

        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, 100),
            _ => panic!("expected Int"),
        }
        match xa.load(1) {
            Some(BorrowEntry::Pointer(val)) => assert_eq!(*val, 42u64),
            _ => panic!("expected Pointer"),
        }
    }

    #[test]
    fn mixed_values_and_pointers() {
        for_each_xarray!(ptr, mixed_values_and_pointers_impl);
    }

    // Drops a tree with mixed Int (even indices) and Pointer (odd indices) entries. Correctness
    // depends on KASAN/kmemleak detecting leaks or double-frees.
    fn drop_frees_pointers_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let count = SIZE * 2;
        for i in 0..count {
            if i % 2 == 0 {
                xa.store(i, Entry::try_int(i * 10).unwrap(), flags::GFP_KERNEL)
                    .unwrap();
            } else {
                let b = KBox::new(i as u64, flags::GFP_KERNEL).unwrap();
                xa.store(i, Entry::Pointer(b), flags::GFP_KERNEL).unwrap();
            }
        }
    }

    #[test]
    fn drop_frees_pointers() {
        for_each_xarray!(ptr, drop_frees_pointers_impl);
    }

    // usize::MAX >> 1 is the maximum valid integer entry. Exercises both compile-time validation
    // (Entry::int) and runtime validation (Entry::try_int).
    fn value_encoding_boundary_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        let max = usize::MAX >> 1;

        // Runtime: try_int accepts max, rejects max+1.
        assert!(Entry::<T>::try_int(max).is_some());
        assert!(Entry::<T>::try_int(max + 1).is_none());
        assert!(Entry::<T>::try_int(usize::MAX).is_none());

        // Compile-time: `Entry::int` validates via `const_assert!` in `Bounded::new`.
        xa.store(0, Entry::int::<{ usize::MAX >> 1 }>(), flags::GFP_KERNEL)
            .unwrap();
        match xa.load(0) {
            Some(BorrowEntry::Int(v)) => assert_eq!(v, max),
            _ => panic!("expected max value"),
        };
    }

    #[test]
    fn value_encoding_boundary() {
        for_each_xarray!(value_encoding_boundary_impl);
    }

    fn overwrite_pointer_with_pointer_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let first = KBox::new(42u64, flags::GFP_KERNEL).unwrap();
        xa.store(0, Entry::Pointer(first), flags::GFP_KERNEL)
            .unwrap();

        let second = KBox::new(99u64, flags::GFP_KERNEL).unwrap();
        let old = xa
            .store(0, Entry::Pointer(second), flags::GFP_KERNEL)
            .unwrap();
        match old {
            Some(Entry::Pointer(p)) => assert_eq!(*p, 42u64),
            _ => panic!("expected old Pointer"),
        }
        match xa.load(0) {
            Some(BorrowEntry::Pointer(val)) => assert_eq!(*val, 99u64),
            _ => panic!("expected new Pointer"),
        }
    }

    #[test]
    fn overwrite_pointer_with_pointer() {
        for_each_xarray!(ptr, overwrite_pointer_with_pointer_impl);
    }

    fn erase_at_boundaries_impl<T: ForeignOwnable, const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<T, SHIFT, SIZE>::new();
        xa.store(SIZE - 1, Entry::int::<0xA>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE, Entry::int::<0xB>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE * SIZE - 1, Entry::int::<0xC>(), flags::GFP_KERNEL)
            .unwrap();
        xa.store(SIZE * SIZE, Entry::int::<0xD>(), flags::GFP_KERNEL)
            .unwrap();

        match xa.erase(SIZE * SIZE) {
            Some(Entry::Int(v)) => assert_eq!(v, 0xD),
            _ => panic!("expected erased Int at SIZE*SIZE"),
        }
        match xa.erase(SIZE * SIZE - 1) {
            Some(Entry::Int(v)) => assert_eq!(v, 0xC),
            _ => panic!("expected erased Int at SIZE*SIZE-1"),
        }
        match xa.erase(SIZE) {
            Some(Entry::Int(v)) => assert_eq!(v, 0xB),
            _ => panic!("expected erased Int at SIZE"),
        }
        match xa.erase(SIZE - 1) {
            Some(Entry::Int(v)) => assert_eq!(v, 0xA),
            _ => panic!("expected erased Int at SIZE-1"),
        }
        assert!(xa.is_empty());
    }

    #[test]
    fn erase_at_boundaries() {
        for_each_xarray!(erase_at_boundaries_impl);
    }

    // Erasing the sole pointer must return the owned KBox and free all intermediate nodes.
    fn erase_last_pointer_cascading_impl<const SHIFT: usize, const SIZE: usize>() {
        let mut xa = XArray::<KBox<u64>, SHIFT, SIZE>::new();
        let boxed = KBox::new(777u64, flags::GFP_KERNEL).unwrap();
        xa.store(0, Entry::Pointer(boxed), flags::GFP_KERNEL)
            .unwrap();

        match xa.erase(0) {
            Some(Entry::Pointer(p)) => assert_eq!(*p, 777u64),
            _ => panic!("expected owned Pointer"),
        }
        assert!(xa.is_empty());
    }

    #[test]
    fn erase_last_pointer_cascading() {
        for_each_xarray!(ptr, erase_last_pointer_cascading_impl);
    }
}
