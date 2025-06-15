#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Daniel Gomez <da.gomez@kernel.org>");
MODULE_DESCRIPTION("Test kmemleak by not freeing memory");
MODULE_VERSION("0.1");

static int *leak_var __ro_after_init;

static int __init my_module_init(void)
{
	leak_var = kmalloc(sizeof(int), GFP_KERNEL);
	if (!leak_var) {
		pr_err("Failed to allocate memory\n");
		return -ENOMEM;
	}
	*leak_var = 42;
	pr_info("Memory allocated without freeing for kmemleak test\n");

	return 0;
}

static void __exit my_module_exit(void)
{
	pr_info("Module exiting without freeing memory\n");
	/* Intentionally not calling kfree(leak_var) to check kmemleak */
}

module_init(my_module_init);
module_exit(my_module_exit);
